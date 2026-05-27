import pytest
import torch
import torch.nn.functional as F
import flag_dnn
from tests import accuracy_utils as utils

# one_hot takes LongTensor inputs
SHAPES = [(32,), (1024,), (5333,), (2, 16), (4, 8, 16)]
NUM_CLASSES_LIST = [5, 10, 100, -1]


@pytest.mark.one_hot
@pytest.mark.parametrize("num_classes", NUM_CLASSES_LIST)
@pytest.mark.parametrize("shape", SHAPES)
def test_accuracy_one_hot(shape, num_classes):
    # When num_classes is specified, generate indices within valid range
    if num_classes == -1:
        max_cls = 20
    else:
        max_cls = num_classes
    x = torch.randint(
        0, max_cls, shape, dtype=torch.long, device=flag_dnn.device
    )

    ref_out = F.one_hot(x, num_classes=num_classes)
    with flag_dnn.use_dnn():
        out = F.one_hot(x, num_classes=num_classes)

    utils.gems_assert_equal(out, ref_out)


@pytest.mark.one_hot
def test_accuracy_one_hot_zero_index():
    """Test with class index 0."""
    x = torch.zeros(32, dtype=torch.long, device=flag_dnn.device)
    ref_out = F.one_hot(x, num_classes=10)
    with flag_dnn.use_dnn():
        out = F.one_hot(x, num_classes=10)
    utils.gems_assert_equal(out, ref_out)


@pytest.mark.one_hot
def test_accuracy_one_hot_output_dtype():
    """Output must be LongTensor."""
    x = torch.randint(0, 5, (16,), dtype=torch.long, device=flag_dnn.device)
    with flag_dnn.use_dnn():
        out = F.one_hot(x, num_classes=10)
    assert out.dtype == torch.long


@pytest.mark.one_hot
def test_accuracy_one_hot_auto_num_classes():
    """Test num_classes=-1 inference."""
    x = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long, device=flag_dnn.device)
    ref_out = F.one_hot(x)
    with flag_dnn.use_dnn():
        out = F.one_hot(x)
    assert out.shape == (5, 5)
    utils.gems_assert_equal(out, ref_out)


@pytest.mark.one_hot
def test_accuracy_one_hot_multidim():
    """Test with multi-dimensional input."""
    x = torch.randint(
        0, 8, (2, 4, 6), dtype=torch.long, device=flag_dnn.device
    )
    ref_out = F.one_hot(x, num_classes=8)
    with flag_dnn.use_dnn():
        out = F.one_hot(x, num_classes=8)
    assert out.shape == (2, 4, 6, 8)
    utils.gems_assert_equal(out, ref_out)
