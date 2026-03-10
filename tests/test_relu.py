import pytest
import torch

import flag_dnn

from .accuracy_utils import gems_assert_close


SHAPES = [(32,), (1024,), (5333,), (16384,), (1024 * 1024,)]


@pytest.mark.relu
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("shape", SHAPES)
def test_accuracy_relu(dtype, shape):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device)

    ref_x = x.clone()
    ref_y = torch.relu(ref_x)

    y = flag_dnn.ops.relu(x)

    torch.testing.assert_close(y, ref_y, rtol=1e-5, atol=1e-5)


@pytest.mark.relu
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_accuracy_relu_empty_tensor(dtype):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = torch.randn(0, dtype=dtype, device=flag_dnn.device)

    ref_y = torch.relu(x)

    y = flag_dnn.ops.relu(x)

    assert y.shape == (0,)
    assert y.dtype == dtype
    assert y.device == x.device
    torch.testing.assert_close(y, ref_y, rtol=1e-5, atol=1e-5)


@pytest.mark.relu
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_accuracy_relu_negative_values(dtype):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = torch.randn(100, dtype=dtype, device=flag_dnn.device) - 2.0

    ref_y = torch.relu(x)

    y = flag_dnn.ops.relu(x)

    torch.testing.assert_close(y, ref_y, rtol=1e-5, atol=1e-5)


@pytest.mark.relu
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_accuracy_relu_positive_values(dtype):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = torch.randn(100, dtype=dtype, device=flag_dnn.device) + 2.0

    ref_y = torch.relu(x)

    y = flag_dnn.ops.relu(x)

    torch.testing.assert_close(y, ref_y, rtol=1e-5, atol=1e-5)


@pytest.mark.relu
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_accuracy_relu_mixed_values(dtype):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = torch.randn(100, dtype=dtype, device=flag_dnn.device)

    ref_y = torch.relu(x)

    y = flag_dnn.ops.relu(x)

    torch.testing.assert_close(y, ref_y, rtol=1e-5, atol=1e-5)
