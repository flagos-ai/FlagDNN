import pytest
import torch
import flag_dnn


CUMPROD_CASES = [
    # (shape, dim)
    ((1024,), 0),
    ((2, 3, 4, 5), 3),
    ((2, 3, 4, 5), 0),
    ((2, 3, 4, 5), -2),
    ((128, 5000), 1),  # 长序列考验 FP16/BF16 精度
    ((128, 256), 1),
]


@pytest.mark.cumprod
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16]
)
@pytest.mark.parametrize("shape, dim", CUMPROD_CASES)
def test_accuracy_cumprod(dtype, shape, dim):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    # 防溢出策略
    if dtype.is_floating_point:
        # 围绕 1.0 波动
        x = torch.randn(shape, dtype=dtype, device=flag_dnn.device) * 0.1 + 1.0
    else:
        # 只有 1 和 -1，避开 int64 溢出
        x = torch.randint(0, 2, shape, dtype=dtype, device=flag_dnn.device) * 2 - 1

    # 动态宽容度策略：应对 fp16/bf16 的累乘误差墙
    rtol, atol = 1e-3, 1e-3
    if dtype in (torch.float16, torch.bfloat16):
        d = dim if dim >= 0 else dim + len(shape)
        N = shape[d]
        if N > 1000:
            rtol, atol = 3e-1, 3e-1
        else:
            rtol, atol = 5e-2, 5e-2

    ref_out = torch.cumprod(x, dim=dim)
    with flag_dnn.use_dnn():
        out = torch.cumprod(x, dim=dim)

    if not dtype.is_floating_point:
        torch.testing.assert_close(out, ref_out, rtol=0, atol=0)
    else:
        torch.testing.assert_close(out, ref_out, rtol=rtol, atol=atol)


@pytest.mark.cumprod
def test_accuracy_cumprod_empty():
    x = torch.empty((2, 0, 3), dtype=torch.float32, device=flag_dnn.device)
    ref_out = torch.cumprod(x, dim=1)
    with flag_dnn.use_dnn():
        out = torch.cumprod(x, dim=1)
    torch.testing.assert_close(out, ref_out)
