import pytest
import torch
import torch.nn.functional as F

import flag_dnn
from . import accuracy_utils as utils
from . import conftest as cfg


# (
#   input_shape,
#   weight_shape,
#   has_bias,
#   stride,
#   padding,
#   dilation,
#   groups,
#   unbatched,
#   channels_last,
#   noncontiguous,
# )
CONV3D_CASES = [
    ((1, 2, 5, 6, 7), (4, 2, 3, 3, 3), True, 1, 1, 1, 1, False, False, False),
    ((2, 3, 8, 9, 10), (6, 3, 3, 3, 3), False, (1, 2, 1), 1, 1, 1, False, False, False),
    ((1, 4, 7, 8, 9), (8, 2, 3, 3, 3), True, 1, 1, 1, 2, False, False, False),
    ((1, 4, 6, 7, 8), (4, 1, 3, 3, 3), False, 1, 1, 1, 4, False, False, False),
    ((1, 2, 8, 8, 8), (3, 2, 3, 3, 3), True, 1, "same", 1, 1, False, False, False),
    ((1, 2, 6, 7, 8), (3, 2, 2, 4, 3), False, 1, "same", 1, 1, False, False, False),
    ((1, 2, 8, 9, 10), (3, 2, 3, 3, 3), False, 2, "valid", 1, 1, False, False, False),
    ((2, 4, 5, 6, 7), (5, 4, 1, 1, 1), True, 1, 0, 1, 1, False, True, False),
    ((2, 3, 8, 8, 8), (4, 3, 3, 3, 3), False, 1, 2, 2, 1, False, False, False),
    ((3, 5, 6, 7), (4, 3, 3, 3, 3), True, 1, 1, 1, 1, True, False, False),
    ((2, 3, 6, 7, 8), (4, 3, 3, 3, 3), True, 1, 1, 1, 1, False, False, True),
]

if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
else:
    FLOAT_DTYPES = utils.ALL_FLOAT_DTYPES


def _conv_reduce_dim(weight_shape):
    return max(
        weight_shape[1]
        * weight_shape[2]
        * weight_shape[3]
        * weight_shape[4],
        1,
    )


def _make_tensor(shape, dtype, channels_last=False, noncontiguous=False):
    if channels_last:
        return torch.randn(shape, dtype=dtype, device=flag_dnn.device).contiguous(
            memory_format=torch.channels_last_3d
        )
    if noncontiguous:
        widened = tuple(shape[:-1]) + (shape[-1] * 2,)
        base = torch.randn(widened, dtype=dtype, device=flag_dnn.device)
        return base[..., ::2]
    return torch.randn(shape, dtype=dtype, device=flag_dnn.device)


@pytest.mark.conv3d
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize(
    (
        "input_shape, weight_shape, has_bias, stride, padding, dilation, "
        "groups, unbatched, channels_last, noncontiguous"
    ),
    CONV3D_CASES,
)
def test_accuracy_conv3d(
    dtype,
    input_shape,
    weight_shape,
    has_bias,
    stride,
    padding,
    dilation,
    groups,
    unbatched,
    channels_last,
    noncontiguous,
):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = _make_tensor(input_shape, dtype, channels_last, noncontiguous)
    w = _make_tensor(weight_shape, dtype, False, noncontiguous)
    b = (
        _make_tensor((weight_shape[0] * 2,), dtype)[::2]
        if has_bias and noncontiguous
        else (
            torch.randn(weight_shape[0], dtype=dtype, device=flag_dnn.device)
            if has_bias
            else None
        )
    )

    x_ref = utils.to_reference(x.detach(), ref_kind="compute")
    w_ref = utils.to_reference(w.detach(), ref_kind="compute")
    b_ref = (
        utils.to_reference(b.detach(), ref_kind="compute")
        if b is not None
        else None
    )

    out_ref = F.conv3d(
        x_ref,
        w_ref,
        b_ref,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )

    with flag_dnn.use_dnn(include=["conv3d"]):
        out_custom = F.conv3d(
            x,
            w,
            b,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )

    assert out_custom.dim() == (4 if unbatched else 5)
    utils.gems_assert_close(
        out_custom,
        out_ref,
        dtype,
        reduce_dim=_conv_reduce_dim(weight_shape),
        atol=2e-2,
    )


@pytest.mark.conv3d
def test_conv3d_same_padding_stride_error():
    x = torch.randn((1, 2, 5, 6, 7), dtype=torch.float32, device=flag_dnn.device)
    w = torch.randn((4, 2, 3, 3, 3), dtype=torch.float32, device=flag_dnn.device)
    with flag_dnn.use_dnn(include=["conv3d"]):
        with pytest.raises(RuntimeError):
            F.conv3d(x, w, stride=2, padding="same")


@pytest.mark.conv3d
def test_conv3d_unsupported_complex():
    x = torch.randn((1, 2, 5, 6, 7), dtype=torch.complex64, device=flag_dnn.device)
    w = torch.randn((4, 2, 3, 3, 3), dtype=torch.complex64, device=flag_dnn.device)
    with flag_dnn.use_dnn(include=["conv3d"]):
        with pytest.raises(NotImplementedError):
            F.conv3d(x, w, padding=1)
