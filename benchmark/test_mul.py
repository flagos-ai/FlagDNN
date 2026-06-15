import pytest
import torch

from benchmark import base, consts


@pytest.mark.mul
def test_mul():
    bench = base.BinaryPointwiseBenchmark(
        op_name="mul",
        torch_op=torch.mul,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
