import pytest
import torch
import flag_dnn


SHAPES = [(32,), (1024,), (5333,), (16384,), (1024 * 1024,), (2, 3, 4, 5)]

def _get_non_negative_tensor(shape, dtype, device):
    """
    生成非负张量，避免对负数求平方根产生 NaN，从而破坏容差对比。
    """
    return torch.abs(torch.randn(shape, dtype=dtype, device=device))


@pytest.mark.sqrt
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", SHAPES)
def test_accuracy_sqrt(dtype, shape):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    # 确保输入全为非负数
    x = _get_non_negative_tensor(shape, dtype, flag_dnn.device)

    # 针对不同数据类型动态设置容差 (Tolerance)
    if dtype == torch.bfloat16:
        rtol, atol = 1.6e-2, 1e-2  # BF16 精度极低，需要较宽松的容差
    elif dtype == torch.float16:
        rtol, atol = 1e-3, 1e-3    # FP16 中等宽松
    else:
        rtol, atol = 1e-5, 1e-5    # FP32 和 FP64 保持严格

    ref_out = torch.sqrt(x)
    with flag_dnn.use_dnn():
        out = torch.sqrt(x)

    torch.testing.assert_close(out, ref_out, rtol=rtol, atol=atol)


@pytest.mark.sqrt
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16])
def test_accuracy_sqrt_empty_tensor(dtype):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    # 测试空张量 (shape 为 0)
    x = torch.randn(0, dtype=dtype, device=flag_dnn.device)

    ref_out = torch.sqrt(x)
    with flag_dnn.use_dnn():
        out = torch.sqrt(x)

    assert out.shape == (0,)
    assert out.dtype == dtype
    assert out.device == x.device
    torch.testing.assert_close(out, ref_out, rtol=0, atol=0)