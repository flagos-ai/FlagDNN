import pytest
import torch
import torch.nn.functional as F

from benchmark import base, consts


@pytest.mark.relu6
def test_perf_relu6():
    bench = base.UnaryPointwiseBenchmark(
        op_name="relu6",
        torch_op=F.relu6,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
