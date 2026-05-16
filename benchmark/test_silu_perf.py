import pytest
import torch
import torch.nn.functional as F

from benchmark import base, consts


@pytest.mark.silu
def test_perf_silu():
    bench = base.UnaryPointwiseBenchmark(
        op_name="silu",
        torch_op=F.silu,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
