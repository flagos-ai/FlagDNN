import pytest
import torch
import torch.nn.functional as F

from benchmark import base, consts


def _to_tuple3(val):
    if isinstance(val, int):
        return (val, val, val)
    return tuple(val)


MAX_POOL3D_CASES = [
    ((2, 64, 32, 128, 128), 2, 2, 0),
    ((4, 256, 16, 64, 64), 2, 2, 0),
    ((8, 512, 8, 32, 32), 2, 2, 0),
    ((8, 64, 16, 112, 112), (1, 2, 2), (1, 2, 2), 0),
    ((8, 128, 16, 56, 56), (2, 2, 2), (2, 2, 2), 0),
    ((1, 32, 15, 55, 55), 3, 2, 1),
    ((16, 16, 8, 112, 112), (3, 3, 3), 1, 1),
]


def max_pool3d_input_fn(case, dtype, device, max_peak_bytes):
    shape, kernel_size, stride, padding = case
    element_size = torch.empty((), dtype=dtype).element_size()
    if torch.Size(shape).numel() * element_size > max_peak_bytes:
        return
    x = torch.randn(shape, dtype=dtype, device=device)
    if x.numel() == 0:
        return
    yield x, kernel_size, stride, padding


def max_pool3d_output_numel(args):
    x, kernel_size, stride, padding = args
    kd, kh, kw = _to_tuple3(kernel_size)
    sd, sh, sw = _to_tuple3(stride)
    pd, ph, pw = _to_tuple3(padding)
    depth_out = (x.shape[2] + 2 * pd - kd) // sd + 1
    height_out = (x.shape[3] + 2 * ph - kh) // sh + 1
    width_out = (x.shape[4] + 2 * pw - kw) // sw + 1
    return x.shape[0] * x.shape[1] * depth_out * height_out * width_out


@pytest.mark.max_pool3d
def test_max_pool3d():
    bench = base.PoolingBenchmark(
        op_name="max_pool3d",
        torch_op=F.max_pool3d,
        dtypes=consts.FLOAT_DTYPES,
        input_fn=max_pool3d_input_fn,
        output_numel_fn=max_pool3d_output_numel,
        cases=MAX_POOL3D_CASES,
    )
    bench.run()
