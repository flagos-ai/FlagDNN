import pytest
import torch

from benchmark import base, consts


MV_CASES = [
    (64, 64),
    (256, 256),
    (1024, 1024),
    (4096, 4096),
    (4096, 256),
    (16384, 512),
    (256, 4096),
    (512, 16384),
    (63, 65),
    (127, 255),
    (511, 1000),
    (2048, 768),
    (4096, 1536),
    (8192, 3072),
    (5000, 384),
    (384, 5000),
    (3584, 3584),
    (1892, 3584),
]


def mv_input_fn(shape, dtype, device, max_peak_bytes):
    m, n = shape
    element_size = torch.empty((), dtype=dtype).element_size()
    peak_bytes = (m * n + n + m) * element_size
    if peak_bytes > max_peak_bytes:
        return
    mat = torch.empty((m, n), dtype=dtype, device=device).uniform_(-1.0, 1.0)
    vec = torch.empty((n,), dtype=dtype, device=device).uniform_(-1.0, 1.0)
    yield mat, vec


@pytest.mark.mv
def test_mv():
    bench = base.BlasBenchmark(
        op_name="mv",
        torch_op=torch.mv,
        dtypes=consts.FLOAT_DTYPES,
        input_fn=mv_input_fn,
        cases=MV_CASES,
        max_peak_bytes=6 * 1024**3,
    )
    bench.run()
