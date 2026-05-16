import pytest
import torch
import torch.nn.functional as F

from benchmark import base, consts


MAX_POOL1D_CASES = [
    ((32, 128, 1024), 2, 2, 0),
    ((64, 256, 512), 2, 2, 0),
    ((32, 64, 4096), 3, 1, 1),
    ((16, 512, 1024), 5, 1, 2),
    ((8, 128, 16000), 10, 5, 0),
    ((1, 64, 48000), 100, 50, 0),
    ((16, 3, 1023), 7, 3, 2),
    ((8, 27, 733), 5, 2, 1),
    ((1, 1, 1), 1, 1, 0),
]


def max_pool1d_input_fn(case, dtype, device, max_peak_bytes):
    shape, kernel_size, stride, padding = case
    element_size = torch.empty((), dtype=dtype).element_size()
    if torch.Size(shape).numel() * element_size > max_peak_bytes:
        return
    x = torch.randn(shape, dtype=dtype, device=device)
    if x.numel() == 0:
        return
    yield x, kernel_size, stride, padding


def max_pool1d_output_numel(args):
    x, kernel_size, stride, padding = args
    length_out = (x.shape[-1] + 2 * padding - kernel_size) // stride + 1
    return x.shape[0] * x.shape[1] * length_out


@pytest.mark.max_pool1d
def test_perf_max_pool1d():
    bench = base.PoolingBenchmark(
        op_name="max_pool1d",
        torch_op=F.max_pool1d,
        dtypes=consts.FLOAT_DTYPES,
        input_fn=max_pool1d_input_fn,
        output_numel_fn=max_pool1d_output_numel,
        cases=MAX_POOL1D_CASES,
    )
    bench.run()
