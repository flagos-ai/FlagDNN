import pytest
import torch

from benchmark import base, consts


@pytest.mark.le
def test_perf_le():
    bench = base.BinaryPointwiseBenchmark(
        op_name="le",
        torch_op=torch.le,
        dtypes=consts.FLOAT_DTYPES,
        output_dtype=torch.bool,
    )
    bench.run()
