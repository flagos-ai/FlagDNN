import pytest
import torch.nn.functional as F

from benchmark import base, consts


@pytest.mark.rrelu
def test_rrelu():
    bench = base.UnaryPointwiseWithArgsBenchmark(
        op_name="rrelu",
        torch_op=F.rrelu,
        dtypes=consts.FLOAT_DTYPES,
        extra_args=(1.0 / 8, 1.0 / 3, False),
        input_range=(-1.0, 1.0),
    )
    bench.run()
