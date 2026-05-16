import pytest
import torch

from benchmark import base, consts


@pytest.mark.abs
def test_perf_abs():
    bench = base.UnaryPointwiseBenchmark(
        op_name="abs",
        torch_op=torch.abs,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
