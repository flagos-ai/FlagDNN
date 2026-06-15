import pytest
import torch

from benchmark import base, consts


@pytest.mark.neg
def test_neg():
    bench = base.UnaryPointwiseBenchmark(
        op_name="neg",
        torch_op=torch.neg,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
