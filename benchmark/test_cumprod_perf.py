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


@pytest.mark.cumprod
def test_perf_cumprod():
    bench = base.UnaryDimwiseBenchmark(
        op_name="cumprod",
        torch_op=base.dimwise_dtype_torch_op(torch.cumprod),
        dtypes=consts.FLOAT_DTYPES,
        cases=SCAN_CASES,
        input_range=(0.75, 1.25),
    )
    bench.run()
