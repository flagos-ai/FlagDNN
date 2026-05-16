import pytest
import torch.nn.functional as F

from benchmark import base, consts


@pytest.mark.threshold_
def test_perf_threshold_():
    bench = base.UnaryPointwiseWithArgsBenchmark(
        op_name="threshold_",
        torch_op=F.threshold_,
        dtypes=consts.FLOAT_DTYPES,
        extra_args=(0.0, 0.0),
        input_range=(-1.0, 1.0),
        is_inplace=True,
    )
    bench.run()
