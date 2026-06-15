import pytest
import torch
import torch.nn.functional as F

from benchmark import base, consts


PRELU_CASES = [
    (1024,),
    (65536,),
    (32, 4096),
    (8192, 8192),
    (16, 2048, 4096),
    (8, 32, 2048, 2048),
    (4, 64, 4096, 4096),
    (32, 64, 224, 224),
    (16, 256, 64, 64),
    (8, 1024, 14, 14),
    (4, 1024, 1024, 16),
    (2, 2048, 2048, 16),
]


def prelu_input_fn(shape, dtype, device, max_peak_bytes):
    element_size = torch.empty((), dtype=dtype).element_size()
    if torch.Size(shape).numel() * element_size * 2 > max_peak_bytes:
        return
    x = torch.empty(shape, dtype=dtype, device=device).uniform_(-5.0, 5.0)
    if x.numel() == 0:
        return
    if len(shape) == 1:
        weight = torch.full((), 0.25, dtype=dtype, device=device)
    else:
        weight = torch.full((shape[1],), 0.25, dtype=dtype, device=device)
    yield x, weight


@pytest.mark.prelu
def test_prelu():
    bench = base.UnaryParametricPointwiseBenchmark(
        op_name="prelu",
        torch_op=F.prelu,
        dtypes=consts.FLOAT_DTYPES,
        input_fn=prelu_input_fn,
        cases=PRELU_CASES,
    )
    bench.run()
