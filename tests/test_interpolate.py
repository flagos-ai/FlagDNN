import pytest
import torch
import torch.nn.functional as F
import flag_dnn
from tests import accuracy_utils as utils


# 4D shapes: (N, C, H, W)
SHAPES_4D = [
    (1, 3, 4, 4),
    (2, 16, 8, 8),
    (4, 3, 16, 16),
    (1, 64, 7, 7),
]

# 3D shapes: (N, C, W)
SHAPES_3D = [
    (1, 16, 32),
    (2, 8, 64),
    (4, 4, 128),
]

FLOAT_DTYPES = [torch.float16, torch.float32]


@pytest.mark.interpolate
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("shape", SHAPES_4D)
def test_accuracy_interpolate_nearest_4d_upscale(shape, dtype):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device)
    out_size = (shape[2] * 2, shape[3] * 2)
    ref_x = utils.to_reference(x)
    ref_out = F.interpolate(ref_x, size=out_size, mode="nearest")
    with flag_dnn.use_dnn():
        out = F.interpolate(x, size=out_size, mode="nearest")
    utils.gems_assert_close(out, ref_out, dtype)


@pytest.mark.interpolate
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("shape", SHAPES_4D)
def test_accuracy_interpolate_nearest_4d_downscale(shape, dtype):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device)
    out_size = (max(1, shape[2] // 2), max(1, shape[3] // 2))
    ref_x = utils.to_reference(x)
    ref_out = F.interpolate(ref_x, size=out_size, mode="nearest")
    with flag_dnn.use_dnn():
        out = F.interpolate(x, size=out_size, mode="nearest")
    utils.gems_assert_close(out, ref_out, dtype)


@pytest.mark.interpolate
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("shape", SHAPES_4D)
@pytest.mark.parametrize("align_corners", [True, False])
def test_accuracy_interpolate_bilinear_4d(shape, dtype, align_corners):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device)
    out_size = (shape[2] * 2, shape[3] * 2)
    ref_x = utils.to_reference(x.float())
    ref_out = F.interpolate(
        ref_x, size=out_size, mode="bilinear", align_corners=align_corners
    )

    if dtype == torch.float32:
        rtol, atol = 1e-4, 1e-4
    else:
        rtol, atol = 1e-2, 1e-2

    with flag_dnn.use_dnn():
        out = F.interpolate(
            x, size=out_size, mode="bilinear", align_corners=align_corners
        )
    torch.testing.assert_close(
        out.float(), ref_out.float(), rtol=rtol, atol=atol
    )


@pytest.mark.interpolate
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("shape", SHAPES_3D)
def test_accuracy_interpolate_nearest_3d(shape, dtype):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device)
    out_size = (shape[2] * 2,)
    ref_x = utils.to_reference(x)
    ref_out = F.interpolate(ref_x, size=out_size, mode="nearest")
    with flag_dnn.use_dnn():
        out = F.interpolate(x, size=out_size, mode="nearest")
    utils.gems_assert_close(out, ref_out, dtype)
