import pytest
import torch.nn.functional as F

from benchmark import base, consts


@pytest.mark.selu
def test_perf_selu():
    bench = base.UnaryPointwiseBenchmark(
        op_name="selu",
        torch_op=F.selu,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
