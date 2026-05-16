import pytest
import torch
import torch.nn.functional as F

from benchmark import base, consts


@pytest.mark.softsign
def test_perf_softsign():
    bench = base.UnaryPointwiseBenchmark(
        op_name="softsign",
        torch_op=F.softsign,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
