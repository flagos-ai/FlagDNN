import pytest
import torch

from benchmark import base, consts


@pytest.mark.eq
def test_eq():
    bench = base.BinaryPointwiseBenchmark(
        op_name="eq",
        torch_op=torch.eq,
        dtypes=consts.FLOAT_DTYPES,
        output_dtype=torch.bool,
    )
    bench.run()
