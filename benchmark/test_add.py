import pytest
import torch

from benchmark import base, consts


@pytest.mark.add
def test_add():
    bench = base.BinaryPointwiseBenchmark(
        op_name="add",
        torch_op=torch.add,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
