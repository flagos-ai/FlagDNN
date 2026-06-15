import pytest
import torch

from benchmark import base, consts


PROD_CASES = [
    ((1024 * 1024 * 16,), None, False),
    ((32, 256, 1024), None, False),
    ((1024, 1024 * 16), 1, False),
    ((32, 1024, 1024), 2, False),
    ((8, 128, 4096), 2, False),
    ((1024, 1024), 0, False),
    ((32, 1024, 1024), 0, False),
    ((32, 256, 56, 56), 1, False),
    ((1, 16, 2048, 2048), 2, False),
    ((64, 512, 512), 2, True),
    ((128, 256, 256), 1, True),
]


@pytest.mark.prod
def test_prod():
    bench = base.UnaryReductionBenchmark(
        op_name="prod",
        torch_op=base.reduction_torch_op(torch.prod),
        dtypes=consts.FLOAT_DTYPES,
        cases=PROD_CASES,
        input_range=(-1.0, 1.0),
    )
    bench.run()
