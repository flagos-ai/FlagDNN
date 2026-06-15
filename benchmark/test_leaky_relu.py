import pytest
import torch.nn.functional as F

from benchmark import base, consts


@pytest.mark.leaky_relu
def test_leaky_relu():
    bench = base.UnaryPointwiseWithArgsBenchmark(
        op_name="leaky_relu",
        torch_op=F.leaky_relu,
        dtypes=consts.FLOAT_DTYPES,
        extra_args=(0.01,),
    )
    bench.run()
