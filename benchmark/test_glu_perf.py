import pytest
import torch.nn.functional as F

from benchmark import base, consts


GLU_CASES = [
    ((2,), -1, None),
    ((16,), -1, None),
    ((1024,), -1, None),
    ((65536,), -1, None),
    ((17, 32), -1, None),
    ((1023, 1024), -1, None),
    ((7, 31, 110), -1, None),
    ((32, 128, 768), -1, None),
    ((1, 2048, 4096), -1, None),
    ((2, 3, 32, 64), -1, None),
    ((1, 3, 224, 224), -1, None),
    ((8, 64, 56, 56), -1, None),
    ((16, 128, 28, 28), -1, None),
    ((32, 256, 14, 14), -1, None),
    ((1, 8, 16, 32, 32), -1, None),
]


@pytest.mark.glu
def test_perf_glu():
    bench = base.UnaryDimwiseBenchmark(
        op_name="glu",
        torch_op=base.dimwise_ignore_dtype_torch_op(F.glu),
        dtypes=consts.FLOAT_DTYPES,
        cases=GLU_CASES,
        input_range=(-5.0, 5.0),
        output_factor=0.5,
    )
    bench.run()
