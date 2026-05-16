import pytest
import torch

from benchmark import base, consts


@pytest.mark.tanh
def test_perf_tanh():
    bench = base.UnaryPointwiseBenchmark(
        op_name="tanh",
        torch_op=torch.tanh,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
