import pytest
import torch
import flag_dnn
from tests import accuracy_utils as utils

# (shape, dim, keepdim) test cases
ANY_CASES = [
    # Global reduction
    ((1024,), None, False),
    ((2, 3, 4, 5), None, False),
    # Per-dim reduction
    ((128, 256), 0, False),
    ((128, 256), 1, True),
    ((2, 3, 4, 5), 2, False),
    ((2, 3, 4, 5), -1, True),
]

BOOL_SHAPES = list(utils.POINTWISE_SHAPES) + [(32,), (1024,), (5333,)]


@pytest.mark.any
@pytest.mark.parametrize("shape, dim, keepdim", ANY_CASES)
def test_accuracy_any_float(shape, dim, keepdim):
    x = torch.randn(shape, dtype=torch.float32, device=flag_dnn.device)
    ref_x = utils.to_reference(x)
    if dim is None:
        ref_out = torch.any(ref_x)
        with flag_dnn.use_dnn():
            out = torch.any(x)
    else:
        ref_out = torch.any(ref_x, dim=dim, keepdim=keepdim)
        with flag_dnn.use_dnn():
            out = torch.any(x, dim=dim, keepdim=keepdim)
    utils.gems_assert_equal(out, ref_out)


@pytest.mark.any
@pytest.mark.parametrize("shape", BOOL_SHAPES)
def test_accuracy_any_bool(shape):
    x = torch.randint(0, 2, shape, dtype=torch.bool, device=flag_dnn.device)
    ref_x = utils.to_reference(x)
    ref_out = torch.any(ref_x)
    with flag_dnn.use_dnn():
        out = torch.any(x)
    utils.gems_assert_equal(out, ref_out)


@pytest.mark.any
def test_accuracy_any_all_false():
    """All elements False: any() must return False."""
    x = torch.zeros(1024, dtype=torch.float32, device=flag_dnn.device)
    ref_out = torch.any(x)
    with flag_dnn.use_dnn():
        out = torch.any(x)
    utils.gems_assert_equal(out, ref_out)


@pytest.mark.any
def test_accuracy_any_all_true():
    """All elements True: any() must return True."""
    x = torch.ones(1024, dtype=torch.float32, device=flag_dnn.device)
    ref_out = torch.any(x)
    with flag_dnn.use_dnn():
        out = torch.any(x)
    utils.gems_assert_equal(out, ref_out)


@pytest.mark.any
def test_accuracy_any_dim_keepdim():
    x = torch.randn(4, 8, 16, dtype=torch.float32, device=flag_dnn.device)
    ref_out = torch.any(x, dim=1, keepdim=True)
    with flag_dnn.use_dnn():
        out = torch.any(x, dim=1, keepdim=True)
    assert out.shape == (4, 1, 16)
    utils.gems_assert_equal(out, ref_out)
