import pytest
import torch.nn.functional as F

from benchmark import base, consts


@pytest.mark.gelu
def test_perf_gelu():
    bench = base.UnaryPointwiseBenchmark(
        op_name="gelu",
        torch_op=F.gelu,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
