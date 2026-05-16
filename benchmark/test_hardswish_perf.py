import pytest
import torch
import torch.nn.functional as F

from benchmark import base, consts


@pytest.mark.hardswish
def test_perf_hardswish():
    bench = base.UnaryPointwiseBenchmark(
        op_name="hardswish",
        torch_op=F.hardswish,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
