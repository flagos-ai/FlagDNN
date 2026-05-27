import pytest
import torch.nn.functional as F

from benchmark import base, consts


@pytest.mark.logsigmoid
def test_perf_logsigmoid():
    bench = base.UnaryPointwiseBenchmark(
        op_name="logsigmoid",
        torch_op=F.logsigmoid,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
