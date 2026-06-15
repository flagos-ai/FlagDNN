import pytest
import torch.nn.functional as F

from benchmark import base, consts


@pytest.mark.relu
def test_relu():
    bench = base.UnaryPointwiseBenchmark(
        op_name="relu",
        torch_op=F.relu,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
