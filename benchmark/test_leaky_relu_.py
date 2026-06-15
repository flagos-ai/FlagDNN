import pytest
import torch.nn.functional as F

from benchmark import base, consts


@pytest.mark.leaky_relu_
def test_leaky_relu_():
    bench = base.UnaryPointwiseWithArgsBenchmark(
        op_name="leaky_relu_",
        torch_op=F.leaky_relu_,
        dtypes=consts.FLOAT_DTYPES,
        extra_args=(0.01,),
        is_inplace=True,
    )
    bench.run()
