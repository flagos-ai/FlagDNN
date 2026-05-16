import pytest
import torch
import torch.nn.functional as F

from benchmark import base, consts


ADAPTIVE_MAX_POOL1D_CASES = [
    ((32, 256, 1024), 1),
    ((8, 512, 4096), 1),
    ((32, 128, 1024), 64),
    ((64, 64, 512), 32),
    ((4, 128, 16000), 100),
    ((1, 256, 48000), 256),
    ((16, 3, 1023), 17),
    ((8, 27, 733), 41),
    ((1, 1, 1), 1),
    ((128, 16, 257), 128),
]


def adaptive_max_pool1d_input_fn(case, dtype, device, max_peak_bytes):
    shape, output_size = case
    element_size = torch.empty((), dtype=dtype).element_size()
    if torch.Size(shape).numel() * element_size > max_peak_bytes:
        return
    x = torch.randn(shape, dtype=dtype, device=device)
    if x.numel() == 0:
        return
    yield x, output_size


def adaptive_max_pool1d_output_numel(args):
    x, output_size = args
    if not isinstance(output_size, int):
        output_size = output_size[0]
    return x.shape[0] * x.shape[1] * output_size


@pytest.mark.adaptive_max_pool1d
def test_perf_adaptive_max_pool1d():
    bench = base.PoolingBenchmark(
        op_name="adaptive_max_pool1d",
        torch_op=F.adaptive_max_pool1d,
        dtypes=consts.FLOAT_DTYPES,
        input_fn=adaptive_max_pool1d_input_fn,
        output_numel_fn=adaptive_max_pool1d_output_numel,
        cases=ADAPTIVE_MAX_POOL1D_CASES,
    )
    bench.run()
