from functools import partial

import pytest
import torch
import torch.nn.functional as F

from benchmark import base, consts


BATCH_NORM_CASES = [
    (1024, 256),
    (256, 1024),
    (32, 256, 1024),
    (32, 64, 112, 112),
    (32, 256, 56, 56),
    (32, 512, 28, 28),
    (32, 1024, 14, 14),
    (32, 2048, 7, 7),
    (8, 64, 512, 512),
    (1, 16, 2048, 2048),
    (16, 1024, 64, 64),
    (8, 2048, 64, 64),
]


def batch_norm_input_fn(shape, dtype, device, max_peak_bytes):
    element_size = torch.empty((), dtype=dtype).element_size()
    if torch.Size(shape).numel() * element_size * 2 > max_peak_bytes:
        return
    x = torch.randn(shape, dtype=dtype, device=device)
    if x.numel() == 0:
        return
    num_channels = shape[1] if len(shape) > 1 else shape[0]
    running_mean = torch.zeros(num_channels, dtype=dtype, device=device)
    running_var = torch.ones(num_channels, dtype=dtype, device=device)
    weight = torch.ones(num_channels, dtype=dtype, device=device)
    bias = torch.zeros(num_channels, dtype=dtype, device=device)
    yield x, running_mean, running_var, weight, bias


@pytest.mark.batch_norm
def test_perf_batch_norm():
    bench = base.NormalizationBenchmark(
        op_name="batch_norm",
        torch_op=partial(F.batch_norm, training=True),
        dtypes=consts.FLOAT_DTYPES,
        input_fn=batch_norm_input_fn,
        cases=BATCH_NORM_CASES,
    )
    bench.run()
