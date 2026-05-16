import pytest
import torch
import torch.nn.functional as F

from benchmark import base, consts


def _to_tuple2(val):
    if isinstance(val, int):
        return (val, val)
    return tuple(val)


ADAPTIVE_MAX_POOL2D_CASES = [
    ((32, 2048, 7, 7), (1, 1)),
    ((128, 512, 14, 14), (1, 1)),
    ((16, 256, 112, 112), (56, 56)),
    ((32, 128, 64, 64), (32, 32)),
    ((8, 32, 128, 256), (32, 64)),
    ((4, 32, 1080, 1920), (270, 480)),
    ((16, 16, 223, 225), (17, 19)),
    ((8, 27, 111, 109), (14, 13)),
    ((1, 1, 1, 1), (1, 1)),
    ((64, 16, 15, 15), (7, 5)),
]


def adaptive_max_pool2d_input_fn(case, dtype, device, max_peak_bytes):
    shape, output_size = case
    element_size = torch.empty((), dtype=dtype).element_size()
    if torch.Size(shape).numel() * element_size > max_peak_bytes:
        return
    x = torch.randn(shape, dtype=dtype, device=device)
    if x.numel() == 0:
        return
    yield x, output_size


def adaptive_max_pool2d_output_numel(args):
    x, output_size = args
    out_h, out_w = _to_tuple2(output_size)
    return x.shape[0] * x.shape[1] * out_h * out_w


@pytest.mark.adaptive_max_pool2d
def test_perf_adaptive_max_pool2d():
    bench = base.PoolingBenchmark(
        op_name="adaptive_max_pool2d",
        torch_op=F.adaptive_max_pool2d,
        dtypes=consts.FLOAT_DTYPES,
        input_fn=adaptive_max_pool2d_input_fn,
        output_numel_fn=adaptive_max_pool2d_output_numel,
        cases=ADAPTIVE_MAX_POOL2D_CASES,
    )
    bench.run()
