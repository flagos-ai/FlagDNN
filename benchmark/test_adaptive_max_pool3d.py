import pytest
import torch
import torch.nn.functional as F

from benchmark import base, consts


def _to_tuple3(val):
    if isinstance(val, int):
        return (val, val, val)
    return tuple(val)


ADAPTIVE_MAX_POOL3D_CASES = [
    ((4, 1024, 16, 7, 7), (1, 1, 1)),
    ((8, 512, 8, 14, 14), (1, 1, 1)),
    ((2, 256, 32, 56, 56), (16, 28, 28)),
    ((4, 128, 16, 64, 64), (8, 32, 32)),
    ((1, 64, 64, 128, 128), (16, 128, 128)),
    ((1, 32, 128, 256, 256), (32, 64, 64)),
    ((4, 3, 15, 111, 109), (7, 17, 13)),
    ((2, 27, 9, 55, 57), (4, 11, 9)),
    ((1, 1, 1, 1, 1), (1, 1, 1)),
]


def adaptive_max_pool3d_input_fn(case, dtype, device, max_peak_bytes):
    shape, output_size = case
    element_size = torch.empty((), dtype=dtype).element_size()
    if torch.Size(shape).numel() * element_size > max_peak_bytes:
        return
    x = torch.randn(shape, dtype=dtype, device=device)
    if x.numel() == 0:
        return
    yield x, output_size


def adaptive_max_pool3d_output_numel(args):
    x, output_size = args
    out_d, out_h, out_w = _to_tuple3(output_size)
    return x.shape[0] * x.shape[1] * out_d * out_h * out_w


@pytest.mark.adaptive_max_pool3d
def test_adaptive_max_pool3d():
    bench = base.PoolingBenchmark(
        op_name="adaptive_max_pool3d",
        torch_op=F.adaptive_max_pool3d,
        dtypes=consts.FLOAT_DTYPES,
        input_fn=adaptive_max_pool3d_input_fn,
        output_numel_fn=adaptive_max_pool3d_output_numel,
        cases=ADAPTIVE_MAX_POOL3D_CASES,
    )
    bench.run()
