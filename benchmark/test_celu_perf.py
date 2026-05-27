import pytest
import torch.nn.functional as F

from benchmark import base, consts


@pytest.mark.celu
def test_perf_celu():
    bench = base.UnaryPointwiseWithArgsBenchmark(
        op_name="celu",
        torch_op=F.celu,
        dtypes=consts.FLOAT_DTYPES,
        extra_args=(1.0,),
        input_range=(-5.0, 5.0),
    )
    bench.run()
