import pytest
import torch

from benchmark import base, consts


@pytest.mark.pow
def test_pow():
    bench = base.BinaryPointwiseBenchmark(
        op_name="pow",
        torch_op=torch.pow,
        dtypes=consts.FLOAT_DTYPES,
        lhs_range=(0.1, 1.1),
        rhs_range=(-2.0, 3.0),
        extra_shapes=[
            ((1024, 1024), (1,)),
            ((32, 256, 1024), (1,)),
        ],
    )
    bench.run()
