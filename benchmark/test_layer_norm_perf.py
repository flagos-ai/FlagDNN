import pytest
import torch
import torch.nn.functional as F

from benchmark import base, consts


LAYER_NORM_CASES = [
    ((32, 1024, 1024), (1024,)),
    ((16, 4096, 4096), (4096,)),
    ((8, 8192, 4096), (4096,)),
    ((128, 197, 768), (768,)),
    ((64, 1370, 1024), (1024,)),
    ((32, 256, 56, 56), (256, 56, 56)),
    ((32, 256, 56, 56), (56, 56)),
    ((128, 1, 4096), (4096,)),
    ((256, 1, 2048), (2048,)),
    ((32, 1, 8192), (8192,)),
    ((4, 8192, 2048), (2048,)),
    ((2, 32768, 4096), (4096,)),
    ((1, 128000, 4096), (4096,)),
    ((8, 4096, 8192), (8192,)),
    ((16, 256, 1152), (1152,)),
    ((2048, 49, 96), (96,)),
    ((32, 56, 56, 96), (96,)),
    ((16, 1500, 512), (512,)),
    ((8, 3000, 1280), (1280,)),
    ((1, 512, 256), (256,)),
    ((1, 1, 256), (256,)),
]


def layer_norm_input_fn(case, dtype, device, max_peak_bytes):
    shape, normalized_shape = case
    element_size = torch.empty((), dtype=dtype).element_size()
    if torch.Size(shape).numel() * element_size * 2 > max_peak_bytes:
        return
    x = torch.randn(shape, dtype=dtype, device=device)
    if x.numel() == 0:
        return
    yield x, tuple(normalized_shape)


@pytest.mark.layer_norm
def test_perf_layer_norm():
    bench = base.NormalizationBenchmark(
        op_name="layer_norm",
        torch_op=F.layer_norm,
        dtypes=consts.FLOAT_DTYPES,
        input_fn=layer_norm_input_fn,
        cases=LAYER_NORM_CASES,
    )
    bench.run()
