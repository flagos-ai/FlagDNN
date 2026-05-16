import pytest
import torch
import torch.nn.functional as F

from benchmark import base, consts


GROUP_NORM_CASES = [
    ((32, 256, 56, 56), 32),
    ((16, 512, 28, 28), 32),
    ((8, 1024, 14, 14), 32),
    ((4, 320, 64, 64), 32),
    ((2, 1280, 16, 16), 32),
    ((32, 128, 1024), 8),
    ((16, 256, 4096), 16),
    ((2, 256, 256, 256), 32),
    ((2, 256, 800, 1088), 32),
    ((1, 256, 128, 128), 32),
    ((2, 320, 128, 128), 32),
    ((1, 2560, 16, 16), 32),
    ((2, 128, 16, 56, 56), 32),
    ((1, 32, 128, 128, 128), 8),
    ((8, 64, 112, 112), 64),
    ((8, 64, 112, 112), 1),
    ((128, 16, 8, 8), 4),
]


def group_norm_input_fn(case, dtype, device, max_peak_bytes):
    shape, num_groups = case
    element_size = torch.empty((), dtype=dtype).element_size()
    if torch.Size(shape).numel() * element_size * 2 > max_peak_bytes:
        return
    x = torch.randn(shape, dtype=dtype, device=device)
    if x.numel() == 0:
        return
    yield x, num_groups


@pytest.mark.group_norm
def test_perf_group_norm():
    bench = base.NormalizationBenchmark(
        op_name="group_norm",
        torch_op=F.group_norm,
        dtypes=consts.FLOAT_DTYPES,
        input_fn=group_norm_input_fn,
        cases=GROUP_NORM_CASES,
    )
    bench.run()
