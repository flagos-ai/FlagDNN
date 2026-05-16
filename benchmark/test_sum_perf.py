import pytest
import torch

from benchmark import base, consts


@pytest.mark.sum
def test_perf_sum():
    bench = base.UnaryReductionBenchmark(
        op_name="sum",
        torch_op=base.reduction_torch_op(torch.sum),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
