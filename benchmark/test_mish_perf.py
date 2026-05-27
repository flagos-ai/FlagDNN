import pytest
import torch.nn.functional as F

from benchmark import base, consts


@pytest.mark.mish
def test_perf_mish():
    bench = base.UnaryPointwiseBenchmark(
        op_name="mish",
        torch_op=F.mish,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
