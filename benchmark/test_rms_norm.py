import pytest
import torch
import torch.nn.functional as F

from benchmark import base, consts


RMS_NORM_CASES = [
    ((256, 1, 2048), (2048,)),
    ((128, 1, 4096), (4096,)),
    ((32, 1, 8192), (8192,)),
    ((1, 1, 4096), (4096,)),
    ((16384, 4096), (4096,)),
    ((65536, 4096), (4096,)),
    ((100000, 8192), (8192,)),
    ((16, 2048, 1536), (1536,)),
    ((16, 2048, 3072), (3072,)),
    ((8, 4096, 5120), (5120,)),
    ((8, 4096, 3584), (3584,)),
    ((1, 512, 1024), (1024,)),
    ((1, 1, 1024), (1024,)),
]


def rms_norm_input_fn(case, dtype, device, max_peak_bytes):
    shape, normalized_shape = case
    element_size = torch.empty((), dtype=dtype).element_size()
    if torch.Size(shape).numel() * element_size * 2 > max_peak_bytes:
        return
    x = torch.randn(shape, dtype=dtype, device=device)
    if x.numel() == 0:
        return
    yield x, tuple(normalized_shape)


@pytest.mark.rms_norm
def test_rms_norm():
    bench = base.NormalizationBenchmark(
        op_name="rms_norm",
        torch_op=F.rms_norm,
        dtypes=consts.FLOAT_DTYPES,
        input_fn=rms_norm_input_fn,
        cases=RMS_NORM_CASES,
    )
    bench.run()
