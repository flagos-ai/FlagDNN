import pytest
import torch
import torch.nn.functional as F

from benchmark import base, consts


@pytest.mark.threshold
def test_perf_threshold():
    bench = base.UnaryPointwiseWithArgsBenchmark(
        op_name="threshold",
        torch_op=F.threshold,
        dtypes=consts.FLOAT_DTYPES,
        extra_args=(0.0, 0.0),
        input_range=(-1.0, 1.0),
    )
    bench.run()
