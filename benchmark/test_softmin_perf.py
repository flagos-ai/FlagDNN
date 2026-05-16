import pytest
import torch.nn.functional as F

from benchmark import base, consts


SOFTMIN_CASES = [
    ((1024,), -1, None),
    ((1024,), 0, None),
    ((17, 31), -1, None),
    ((17, 31), 0, None),
    ((17, 31), 1, None),
    ((256, 1000), -1, None),
    ((256, 1000), 0, None),
    ((256, 1000), 1, None),
    ((32, 4096), -1, None),
    ((32, 4096), 0, None),
    ((32, 4096), 1, None),
    ((7, 31, 109), -1, None),
    ((7, 31, 109), 0, None),
    ((7, 31, 109), 1, None),
    ((4, 512, 1024), -1, None),
    ((8, 128, 4096), -1, None),
    ((2, 3, 32, 32), -1, None),
    ((2, 3, 32, 32), 0, None),
    ((2, 3, 32, 32), 1, None),
    ((8, 64, 56, 56), -1, None),
    ((16, 128, 28, 28), -1, None),
    ((2, 12, 512, 512), -1, None),
]


@pytest.mark.softmin
def test_perf_softmin():
    bench = base.UnaryDimwiseBenchmark(
        op_name="softmin",
        torch_op=base.dimwise_dtype_torch_op(F.softmin),
        dtypes=consts.FLOAT_DTYPES,
        cases=SOFTMIN_CASES,
        input_range=(-5.0, 5.0),
    )
    bench.run()
