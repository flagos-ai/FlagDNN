import pytest
import torch

from benchmark import base, consts


MEAN_CASES = [
    ((1024 * 1024 * 16,), None, False),
    ((32, 256, 1024), None, False),
    ((1024, 1024 * 16), 1, False),
    ((32, 1024, 1024), 2, False),
    ((8, 128, 4096), 2, False),
    ((1024 * 16, 1024), 0, False),
    ((32, 1024, 1024), 0, False),
    ((32, 256, 56, 56), (2, 3), False),
    ((32, 256, 56, 56), 1, False),
    ((1, 16, 2048, 2048), (2, 3), False),
    ((64, 512, 512), 2, True),
    ((128, 256, 256), 1, True),
]


@pytest.mark.mean
def test_perf_mean():
    bench = base.UnaryReductionBenchmark(
        op_name="mean",
        torch_op=base.reduction_torch_op(torch.mean),
        dtypes=consts.FLOAT_DTYPES,
        cases=MEAN_CASES,
    )
    bench.run()
