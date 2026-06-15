import pytest
import torch.nn.functional as F

from benchmark import base, consts


@pytest.mark.elu
def test_elu():
    bench = base.UnaryPointwiseWithArgsBenchmark(
        op_name="elu",
        torch_op=F.elu,
        dtypes=consts.FLOAT_DTYPES,
        extra_args=(1.0,),
        input_range=(-3.0, 3.0),
    )
    bench.run()
