import pytest
import torch
import torch.nn.functional as F

from benchmark import base, consts


def _to_tuple2(val):
    if isinstance(val, int):
        return (val, val)
    return tuple(val)


MAX_POOL2D_CASES = [
    ((128, 64, 56, 56), 2, 2, 0),
    ((64, 128, 56, 56), 2, 2, 0),
    ((32, 256, 14, 14), 2, 2, 0),
    ((16, 512, 7, 7), 2, 2, 0),
    ((64, 64, 112, 112), 3, 2, 1),
    ((64, 192, 28, 28), 3, 1, 1),
    ((32, 256, 28, 28), 3, 1, 1),
    ((8, 64, 800, 800), 2, 2, 0),
    ((4, 128, 1080, 1920), 2, 2, 0),
    ((32, 1, 128, 256), (2, 3), (2, 2), (0, 1)),
]


def max_pool2d_input_fn(case, dtype, device, max_peak_bytes):
    shape, kernel_size, stride, padding = case
    element_size = torch.empty((), dtype=dtype).element_size()
    if torch.Size(shape).numel() * element_size > max_peak_bytes:
        return
    x = torch.randn(shape, dtype=dtype, device=device)
    if x.numel() == 0:
        return
    yield x, kernel_size, stride, padding


def max_pool2d_output_numel(args):
    x, kernel_size, stride, padding = args
    kh, kw = _to_tuple2(kernel_size)
    sh, sw = _to_tuple2(stride)
    ph, pw = _to_tuple2(padding)
    height_out = (x.shape[2] + 2 * ph - kh) // sh + 1
    width_out = (x.shape[3] + 2 * pw - kw) // sw + 1
    return x.shape[0] * x.shape[1] * height_out * width_out


@pytest.mark.max_pool2d
def test_perf_max_pool2d():
    bench = base.PoolingBenchmark(
        op_name="max_pool2d",
        torch_op=F.max_pool2d,
        dtypes=consts.FLOAT_DTYPES,
        input_fn=max_pool2d_input_fn,
        output_numel_fn=max_pool2d_output_numel,
        cases=MAX_POOL2D_CASES,
    )
    bench.run()
