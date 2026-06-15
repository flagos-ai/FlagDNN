import pytest
import torch

from benchmark import base, consts


@pytest.mark.gt
def test_gt():
    bench = base.BinaryPointwiseBenchmark(
        op_name="gt",
        torch_op=torch.gt,
        dtypes=consts.FLOAT_DTYPES,
        output_dtype=torch.bool,
    )
    bench.run()
