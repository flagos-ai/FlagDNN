import pytest
import torch.nn.functional as F

from benchmark import base, consts


@pytest.mark.softplus
def test_softplus():
    bench = base.UnaryPointwiseWithArgsBenchmark(
        op_name="softplus",
        torch_op=F.softplus,
        dtypes=consts.FLOAT_DTYPES,
        extra_arg_cases=[
            (1.0, 20.0),
            (0.5, 20.0),
            (2.0, 20.0),
            (1.0, 10.0),
            (1.0, 30.0),
        ],
        input_range=(-10.0, 10.0),
    )
    bench.run()
