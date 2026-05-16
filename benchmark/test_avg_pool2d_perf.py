import pytest
import torch
import torch.nn.functional as F

from benchmark import base, consts


def _to_tuple2(val):
    if isinstance(val, int):
        return (val, val)
    return tuple(val)


AVG_POOL2D_CASES = [
    ((128, 64, 56, 56), 2, 2, 0),
    ((64, 128, 56, 56), 2, 2, 0),
    ((128, 512, 7, 7), 7, 1, 0),
    ((64, 1024, 7, 7), 7, 1, 0),
    ((32, 1280, 7, 7), 7, 1, 0),
    ((16, 1536, 10, 10), 10, 1, 0),
    ((32, 128, 56, 56), 2, 2, 0),
    ((16, 256, 56, 56), 2, 2, 0),
    ((4, 32, 1024, 1024), 4, 4, 0),
    ((8, 64, 256, 1024), 2, 2, 0),
]


def avg_pool2d_input_fn(case, dtype, device, max_peak_bytes):
    shape, kernel_size, stride, padding = case
    element_size = torch.empty((), dtype=dtype).element_size()
    if torch.Size(shape).numel() * element_size > max_peak_bytes:
        return
    x = torch.randn(shape, dtype=dtype, device=device)
    if x.numel() == 0:
        return
    yield x, kernel_size, stride, padding


def avg_pool2d_output_numel(args):
    x, kernel_size, stride, padding = args
    kh, kw = _to_tuple2(kernel_size)
    sh, sw = _to_tuple2(stride)
    ph, pw = _to_tuple2(padding)
    height_out = (x.shape[2] + 2 * ph - kh) // sh + 1
    width_out = (x.shape[3] + 2 * pw - kw) // sw + 1
    return x.shape[0] * x.shape[1] * height_out * width_out


@pytest.mark.avg_pool2d
def test_perf_avg_pool2d():
    bench = base.PoolingBenchmark(
        op_name="avg_pool2d",
        torch_op=F.avg_pool2d,
        dtypes=consts.FLOAT_DTYPES,
        input_fn=avg_pool2d_input_fn,
        output_numel_fn=avg_pool2d_output_numel,
        cases=AVG_POOL2D_CASES,
    )
    bench.run()
