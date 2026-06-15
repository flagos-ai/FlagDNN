import pytest
import torch

from benchmark import base, consts


CLAMP_CASES = [
    ((1024 * 256,), None, None),
    ((32, 256, 56, 56), None, None),
    ((2, 8192, 8192), None, None),
    ((127,), (127,), (127,)),
    ((1024 * 256,), (1024 * 256,), (1024 * 256,)),
    ((1, 2048, 4096), (1, 2048, 4096), (1, 2048, 4096)),
    ((32, 256, 56, 56), (256, 1, 1), (256, 1, 1)),
    ((16, 1024, 14, 14), (1024, 1, 1), (1024, 1, 1)),
    ((8, 128, 12288), (12288,), (12288,)),
    ((1024, 1024), (1,), (1,)),
]


def clamp_input_fn(case, dtype, device, max_peak_bytes):
    shape_x, shape_min, shape_max = case
    element_size = torch.empty((), dtype=dtype).element_size()
    bytes_x = torch.Size(shape_x).numel() * element_size
    bytes_min = (
        torch.Size(shape_min).numel() * element_size
        if shape_min is not None
        else 0
    )
    bytes_max = (
        torch.Size(shape_max).numel() * element_size
        if shape_max is not None
        else 0
    )
    if bytes_x * 2 + bytes_min + bytes_max > max_peak_bytes:
        return

    x = torch.empty(shape_x, dtype=dtype, device=device).uniform_(-5.0, 5.0)
    if x.numel() == 0:
        return
    min_val = (
        -2.0
        if shape_min is None
        else torch.empty(shape_min, dtype=dtype, device=device).uniform_(
            -4.0, -1.0
        )
    )
    max_val = (
        2.0
        if shape_max is None
        else torch.empty(shape_max, dtype=dtype, device=device).uniform_(
            1.0, 4.0
        )
    )
    yield x, min_val, max_val


@pytest.mark.clamp
def test_clamp():
    bench = base.UnaryParametricPointwiseBenchmark(
        op_name="clamp",
        torch_op=torch.clamp,
        dtypes=consts.FLOAT_DTYPES,
        input_fn=clamp_input_fn,
        cases=CLAMP_CASES,
    )
    bench.run()
