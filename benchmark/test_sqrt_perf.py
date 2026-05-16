import pytest
import torch

from benchmark import base, consts


@pytest.mark.sqrt
def test_perf_sqrt():
    bench = base.UnaryPointwiseBenchmark(
        op_name="sqrt",
        torch_op=torch.sqrt,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
