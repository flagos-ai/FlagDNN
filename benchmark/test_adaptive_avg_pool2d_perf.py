import pytest
import torch
import torch.nn.functional as F

from benchmark import base, consts


def _to_tuple2(val):
    if isinstance(val, int):
        return (val, val)
    return tuple(val)


ADAPTIVE_AVG_POOL2D_CASES = [
    ((32, 256, 14, 14), 1),
    ((16, 512, 7, 7), (1, 1)),
    ((32, 128, 28, 28), 14),
    ((8, 64, 224, 224), 7),
    ((4, 128, 500, 500), 10),
    ((32, 128, 224, 224), 15),
    ((16, 64, 300, 300), (42, 42)),
    ((16, 3, 224, 224), 1),
    ((32, 27, 112, 112), 16),
    ((8, 128, 223, 223), (14, 14)),
    ((1, 1, 1, 1), 1),
    ((128, 16, 32, 32), 32),
    ((1024, 64, 8, 8), 2),
]


def adaptive_avg_pool2d_input_fn(case, dtype, device, max_peak_bytes):
    shape, output_size = case
    element_size = torch.empty((), dtype=dtype).element_size()
    if torch.Size(shape).numel() * element_size > max_peak_bytes:
        return
    x = torch.randn(shape, dtype=dtype, device=device)
    if x.numel() == 0:
        return
    yield x, output_size


def adaptive_avg_pool2d_output_numel(args):
    x, output_size = args
    out_h, out_w = _to_tuple2(output_size)
    return x.shape[0] * x.shape[1] * out_h * out_w


@pytest.mark.adaptive_avg_pool2d
def test_perf_adaptive_avg_pool2d():
    bench = base.PoolingBenchmark(
        op_name="adaptive_avg_pool2d",
        torch_op=F.adaptive_avg_pool2d,
        dtypes=consts.FLOAT_DTYPES,
        input_fn=adaptive_avg_pool2d_input_fn,
        output_numel_fn=adaptive_avg_pool2d_output_numel,
        cases=ADAPTIVE_AVG_POOL2D_CASES,
    )
    bench.run()
