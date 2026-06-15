import pytest
import torch

from benchmark import base, consts


@pytest.mark.ge
def test_ge():
    bench = base.BinaryPointwiseBenchmark(
        op_name="ge",
        torch_op=torch.ge,
        dtypes=consts.FLOAT_DTYPES,
        output_dtype=torch.bool,
    )
    bench.run()
