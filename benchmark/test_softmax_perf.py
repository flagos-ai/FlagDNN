import pytest
import torch.nn.functional as F

from benchmark import base, consts


@pytest.mark.softmax
def test_perf_softmax():
    bench = base.UnaryDimwiseBenchmark(
        op_name="softmax",
        torch_op=base.dimwise_dtype_torch_op(F.softmax),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
