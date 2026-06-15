import pytest
import torch.nn.functional as F

from benchmark import base, consts


@pytest.mark.hardtanh_
def test_hardtanh_():
    bench = base.UnaryPointwiseWithArgsBenchmark(
        op_name="hardtanh_",
        torch_op=F.hardtanh_,
        dtypes=consts.FLOAT_DTYPES,
        extra_args=(-1.0, 1.0),
        input_range=(-2.0, 2.0),
        is_inplace=True,
    )
    bench.run()
