import pytest
import torch.nn.functional as F

from benchmark import base, consts


@pytest.mark.elu_
def test_perf_elu_():
    bench = base.UnaryPointwiseWithArgsBenchmark(
        op_name="elu_",
        torch_op=F.elu_,
        dtypes=consts.FLOAT_DTYPES,
        extra_args=(1.0,),
        input_range=(-3.0, 3.0),
        is_inplace=True,
    )
    bench.run()
