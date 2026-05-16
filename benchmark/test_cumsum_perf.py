import pytest
import torch

from benchmark import base, consts


SCAN_CASES = [
    ((1024,), 0, None),
    ((32, 256, 1024), 2, None),
    ((32, 256, 1024), 0, None),
    ((32, 256, 1024), 1, None),
    ((1024, 1024), 1, None),
    ((1024, 1024), 0, None),
    ((32, 256, 56, 56), 3, None),
    ((32, 256, 56, 56), 1, None),
]


@pytest.mark.cumsum
def test_perf_cumsum():
    bench = base.UnaryDimwiseBenchmark(
        op_name="cumsum",
        torch_op=base.dimwise_dtype_torch_op(torch.cumsum),
        dtypes=consts.FLOAT_DTYPES,
        cases=SCAN_CASES,
    )
    bench.run()
