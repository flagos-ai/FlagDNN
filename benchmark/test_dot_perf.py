import pytest
import torch

from benchmark import base, consts


DOT_CASES = [
    (256,),
    (512,),
    (768,),
    (1024,),
    (2048,),
    (3072,),
    (4096,),
    (6144,),
    (8192,),
    (12288,),
    (24576,),
    (32768,),
    (49152,),
    (65536,),
    (131072,),
    (262144,),
    (524288,),
    (1048576,),
    (2097152,),
    (4194304,),
    (8388608,),
]


def dot_input_fn(shape, dtype, device, max_peak_bytes):
    numel = shape[0]
    element_size = torch.empty((), dtype=dtype).element_size()
    if numel == 0 or numel * element_size * 2 > max_peak_bytes:
        return
    x = torch.empty(shape, dtype=dtype, device=device).uniform_(-1.0, 1.0)
    y = torch.empty(shape, dtype=dtype, device=device).uniform_(-1.0, 1.0)
    yield x, y


@pytest.mark.dot
def test_perf_dot():
    bench = base.BlasBenchmark(
        op_name="dot",
        torch_op=torch.dot,
        dtypes=consts.FLOAT_DTYPES,
        input_fn=dot_input_fn,
        cases=DOT_CASES,
        max_peak_bytes=6 * 1024**3,
    )
    bench.run()
