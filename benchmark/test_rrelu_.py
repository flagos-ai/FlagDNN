import pytest
import torch.nn.functional as F

from benchmark import base, consts


@pytest.mark.rrelu_
def test_rrelu_():
    bench = base.UnaryPointwiseWithArgsBenchmark(
        op_name="rrelu_",
        torch_op=F.rrelu_,
        dtypes=consts.FLOAT_DTYPES,
        extra_args=(1.0 / 8, 1.0 / 3, False),
        input_range=(-1.0, 1.0),
        is_inplace=True,
    )
    bench.run()
