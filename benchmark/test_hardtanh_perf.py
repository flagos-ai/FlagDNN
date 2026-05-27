import pytest
import torch.nn.functional as F

from benchmark import base, consts


@pytest.mark.hardtanh
def test_perf_hardtanh():
    bench = base.UnaryPointwiseWithArgsBenchmark(
        op_name="hardtanh",
        torch_op=F.hardtanh,
        dtypes=consts.FLOAT_DTYPES,
        extra_args=(-1.0, 1.0),
        input_range=(-2.0, 2.0),
    )
    bench.run()
