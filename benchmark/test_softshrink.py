import pytest
import torch.nn.functional as F

from benchmark import base, consts


@pytest.mark.softshrink
def test_softshrink():
    bench = base.UnaryPointwiseWithArgsBenchmark(
        op_name="softshrink",
        torch_op=F.softshrink,
        dtypes=consts.FLOAT_DTYPES,
        extra_arg_cases=[(0.0,), (0.5,), (1.0,), (1.5,)],
        input_range=(-5.0, 5.0),
    )
    bench.run()
