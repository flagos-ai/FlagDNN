import pytest
import torch
import torch.nn.functional as F

from benchmark import base, consts


def _to_tuple3(val):
    if isinstance(val, int):
        return (val, val, val)
    return tuple(val)


AVG_POOL3D_CASES = [
    ((8, 1024, 8, 7, 7), (8, 7, 7), 1, 0),
    ((2, 2048, 16, 4, 4), (16, 4, 4), 1, 0),
    ((1, 512, 32, 8, 8), (32, 8, 8), 1, 0),
    ((4, 128, 16, 64, 64), 2, 2, 0),
    ((8, 64, 32, 112, 112), (1, 2, 2), (1, 2, 2), 0),
    ((4, 256, 128, 14, 14), (4, 1, 1), (4, 1, 1), 0),
    ((2, 64, 16, 32, 32), 3, 1, 1),
]


def avg_pool3d_input_fn(case, dtype, device, max_peak_bytes):
    shape, kernel_size, stride, padding = case
    element_size = torch.empty((), dtype=dtype).element_size()
    if torch.Size(shape).numel() * element_size > max_peak_bytes:
        return
    x = torch.randn(shape, dtype=dtype, device=device)
    if x.numel() == 0:
        return
    yield x, kernel_size, stride, padding


def avg_pool3d_output_numel(args):
    x, kernel_size, stride, padding = args
    kd, kh, kw = _to_tuple3(kernel_size)
    sd, sh, sw = _to_tuple3(stride)
    pd, ph, pw = _to_tuple3(padding)
    depth_out = (x.shape[2] + 2 * pd - kd) // sd + 1
    height_out = (x.shape[3] + 2 * ph - kh) // sh + 1
    width_out = (x.shape[4] + 2 * pw - kw) // sw + 1
    return x.shape[0] * x.shape[1] * depth_out * height_out * width_out


@pytest.mark.avg_pool3d
def test_avg_pool3d():
    bench = base.PoolingBenchmark(
        op_name="avg_pool3d",
        torch_op=F.avg_pool3d,
        dtypes=consts.FLOAT_DTYPES,
        input_fn=avg_pool3d_input_fn,
        output_numel_fn=avg_pool3d_output_numel,
        cases=AVG_POOL3D_CASES,
    )
    bench.run()
