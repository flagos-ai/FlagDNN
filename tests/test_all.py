import pytest
import torch
import flag_dnn
from tests import accuracy_utils as utils

ALL_CASES = [
    ((1024,), None, False),
    ((2, 3, 4, 5), None, False),
    ((128, 256), 0, False),
    ((128, 256), 1, True),
    ((2, 3, 4, 5), 2, False),
    ((2, 3, 4, 5), -1, True),
]

BOOL_SHAPES = list(utils.POINTWISE_SHAPES) + [(32,), (1024,), (5333,)]


@pytest.mark.all
@pytest.mark.parametrize("shape, dim, keepdim", ALL_CASES)
def test_accuracy_all_float(shape, dim, keepdim):
    x = torch.randn(shape, dtype=torch.float32, device=flag_dnn.device)
    ref_x = utils.to_reference(x)
    if dim is None:
        ref_out = torch.all(ref_x)
        with flag_dnn.use_dnn():
            out = torch.all(x)
    else:
        ref_out = torch.all(ref_x, dim=dim, keepdim=keepdim)
        with flag_dnn.use_dnn():
            out = torch.all(x, dim=dim, keepdim=keepdim)
    utils.gems_assert_equal(out, ref_out)


@pytest.mark.all
@pytest.mark.parametrize("shape", BOOL_SHAPES)
def test_accuracy_all_bool(shape):
    x = torch.randint(0, 2, shape, dtype=torch.bool, device=flag_dnn.device)
    ref_x = utils.to_reference(x)
    ref_out = torch.all(ref_x)
    with flag_dnn.use_dnn():
        out = torch.all(x)
    utils.gems_assert_equal(out, ref_out)


@pytest.mark.all
def test_accuracy_all_all_true():
    """All nonzero elements: all() must return True."""
    x = torch.ones(1024, dtype=torch.float32, device=flag_dnn.device)
    ref_out = torch.all(x)
    with flag_dnn.use_dnn():
        out = torch.all(x)
    utils.gems_assert_equal(out, ref_out)


@pytest.mark.all
def test_accuracy_all_one_false():
    """One zero element: all() must return False."""
    x = torch.ones(1024, dtype=torch.float32, device=flag_dnn.device)
    x[500] = 0.0
    ref_out = torch.all(x)
    with flag_dnn.use_dnn():
        out = torch.all(x)
    utils.gems_assert_equal(out, ref_out)


@pytest.mark.all
def test_accuracy_all_dim_keepdim():
    x = torch.randn(4, 8, 16, dtype=torch.float32, device=flag_dnn.device)
    ref_out = torch.all(x, dim=1, keepdim=True)
    with flag_dnn.use_dnn():
        out = torch.all(x, dim=1, keepdim=True)
    assert out.shape == (4, 1, 16)
    utils.gems_assert_equal(out, ref_out)
