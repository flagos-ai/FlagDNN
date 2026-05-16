import pytest
import torch
import torch.nn.functional as F

from benchmark import base, consts


def _to_tuple3(val):
    if isinstance(val, int):
        return (val, val, val)
    return tuple(val)


ADAPTIVE_AVG_POOL3D_CASES = [
    ((4, 256, 16, 14, 14), 1),
    ((2, 512, 8, 7, 7), (1, 1, 1)),
    ((4, 128, 16, 28, 28), 14),
    ((2, 64, 32, 112, 112), (8, 7, 7)),
    ((4, 128, 16, 112, 112), 15),
    ((2, 64, 10, 150, 150), (4, 42, 42)),
    ((4, 3, 16, 112, 112), 1),
    ((8, 27, 8, 56, 56), 7),
    ((2, 128, 15, 111, 111), (7, 14, 14)),
    ((1, 1, 1, 1, 1), 1),
    ((32, 16, 8, 16, 16), 8),
    ((128, 64, 4, 4, 4), 2),
]


def adaptive_avg_pool3d_input_fn(case, dtype, device, max_peak_bytes):
    shape, output_size = case
    element_size = torch.empty((), dtype=dtype).element_size()
    if torch.Size(shape).numel() * element_size > max_peak_bytes:
        return
    x = torch.randn(shape, dtype=dtype, device=device)
    if x.numel() == 0:
        return
    yield x, output_size


def adaptive_avg_pool3d_output_numel(args):
    x, output_size = args
    out_d, out_h, out_w = _to_tuple3(output_size)
    return x.shape[0] * x.shape[1] * out_d * out_h * out_w


@pytest.mark.adaptive_avg_pool3d
def test_perf_adaptive_avg_pool3d():
    bench = base.PoolingBenchmark(
        op_name="adaptive_avg_pool3d",
        torch_op=F.adaptive_avg_pool3d,
        dtypes=consts.FLOAT_DTYPES,
        input_fn=adaptive_avg_pool3d_input_fn,
        output_numel_fn=adaptive_avg_pool3d_output_numel,
        cases=ADAPTIVE_AVG_POOL3D_CASES,
    )
    bench.run()
