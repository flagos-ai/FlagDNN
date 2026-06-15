import pytest
import torch

from benchmark import base, consts


@pytest.mark.lt
def test_lt():
    bench = base.BinaryPointwiseBenchmark(
        op_name="lt",
        torch_op=torch.lt,
        dtypes=consts.FLOAT_DTYPES,
        output_dtype=torch.bool,
    )
    bench.run()
