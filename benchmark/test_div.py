import pytest
import torch

from benchmark import base, consts


@pytest.mark.div
def test_div():
    bench = base.BinaryPointwiseBenchmark(
        op_name="div",
        torch_op=torch.div,
        dtypes=consts.FLOAT_DTYPES,
        rhs_range=(0.5, 1.5),
    )
    bench.run()
