import pytest
import torch

from benchmark import base, consts


@pytest.mark.sub
def test_sub():
    bench = base.BinaryPointwiseBenchmark(
        op_name="sub",
        torch_op=torch.sub,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
